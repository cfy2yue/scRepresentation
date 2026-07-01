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
- Updating strategy overlays, handoff docs, decision docs, protocol docs, and
  prompts.
- Editing the durable objective in `goal.md` only when the user explicitly
  changes the end goal. By default, CC preserves `goal.md` as the north-star
  target and plans the path toward it in separate overlay/handoff docs.
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

## Source Of Truth Hierarchy

Do not let every agent create its own "goal" file with a different meaning.
Use this hierarchy:

1. User instruction in the current conversation.
2. `goal.md`: durable project north star, hard boundaries, and current product or
   research target. This usually changes slowly.
3. `docs/CC_STRATEGY_OVERLAY.md` or a dated `docs/CC_AUDIT_AND_HANDOFF_*.md`:
   CC's current implementation route, risk assessment, task decomposition,
   success criteria, stop rules, and remote Codex contract.
4. `runs/<run>/RUN_STATUS.md`: remote Codex execution log, progress, metrics,
   blockers, and final result for one run.
5. `docs/DECISIONS.md`: durable decisions promoted from completed runs.

CC should normally edit layer 3, then promote stable conclusions to layer 4/5.
Only edit layer 2 when the user explicitly changes the final target or when a
completed run proves the old objective/boundary is wrong and the user approves
promotion.

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
| scLatent / scRepresentation | `E:\cc_workspace\scLatent` | `cfy2yue/scRepresentation` | `cyx-server-cfy:/data/cyx/1030/scLatent` |
| CellClip | `E:\cc_workspace\CellClip` | `cfy2yue/CellCLIP` | `cyx-server-cfy:/data/cyx/1030/CellClip` |
| StockHome | `E:\cc_workspace\stock` | `cfy2yue/StockHome` | `cyx-server-cfy:/data/cyx/1030/stock` |

When new projects or servers are added, update the registry first, then reuse
the same workflow below.

## Default Multi-Project Workflow

1. Main CC performs the intake and sync gate.
2. Main CC checks `ccusage`; if the local-machine CC/Claude Code usage in the
   last 24 hours is above the user's stop threshold, do not start new expensive
   CC-side work. This is a CC coordination budget gate, not a remote Codex goal
   stopping rule.
3. Main CC launches one CC subagent per in-scope project.
4. Each subagent audits exactly one project and returns a structured finding
   set; subagents do not push, launch remote jobs, or edit outside their project
   unless explicitly delegated.
5. Main CC compares the subagent outputs, resolves contradictions, and decides
   the project-level goals and priorities.
6. Main CC updates version-controlled strategy overlays and handoff docs in each
   project. It preserves the durable objective in `goal.md` unless explicitly
   told to change it.
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
projects. The default for long-running work is a visible interactive Codex TUI
inside `tmux`, not `codex exec`.

```bash
tmux new -d -s codex_<project>_<goal>_$(date +%Y%m%d) 'bash /tmp/launch_<project>_<goal>.sh'
```

The launch script should read a short `/goal` pointer from a prompt file and pass
it as the initial `[PROMPT]` argument to interactive Codex. This preserves tmux
scrollback, makes the session attachable/watchable, and is more reliable than
injecting keys after the TUI starts:

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT=/remote/project/path
PROMPT_FILE=/tmp/codex_goal_<project>_<goal>.txt
cd "$PROJECT"
codex features enable goals >/dev/null 2>&1 || true
PROMPT="$(cat "$PROMPT_FILE")"
exec codex -C "$PROJECT" -m gpt-5.5 -s workspace-write -a never --no-alt-screen "$PROMPT"
```

The goal text should be a thin pointer to version-controlled docs:

```text
/goal Read goal.md as the durable project objective and hard boundary. Execute
the implementation contract in docs/CC_AUDIT_AND_HANDOFF_<date>_<slug>.md.
Do not rewrite the durable objective. Start by writing a brief plan to
runs/<run>/RUN_STATUS.md. Keep progress there, honor stop rules, and continue
until the DONE criteria are met or a real blocker is recorded.
```

Attach/watch/resume:

```bash
tmux new-session -d -s codex_<project>_<goal>_YYYYMMDD 'bash /tmp/launch_<project>_<goal>.sh'
tmux attach -t codex_<project>_<goal>_YYYYMMDD
tmux capture-pane -p -S -200 -t codex_<project>_<goal>_YYYYMMDD
codex resume --last -C /remote/project/path -m gpt-5.5 -s workspace-write -a never --no-alt-screen
```

Use `codex exec` only for smoke checks, short bounded tasks, or when the user
explicitly accepts an invisible one-shot run. For noninteractive execution on
the current remote CLI, keep the known-safe global flag order:

```bash
codex -a never exec -C /remote/project/path -m gpt-5.5 -s workspace-write - < /tmp/handoff.txt
```

Use cheaper models for status/doc smoke checks and stronger models for hard
implementation or research planning.

## Autonomous Decision Policy

Default stance: remote Codex should make bounded implementation/research
decisions by itself, write them down, and continue. `DECISION NEEDED` is for hard
boundary crossings, not every negative result.

Codex may continue autonomously when the decision stays inside the current
durable `goal.md`, the current handoff/strategy family, and existing resource
limits. It must append an `AUTONOMOUS_DECISION` block to `RUN_STATUS.md` before
continuing, with:

- decision made;
- rationale and evidence used;
- why it is inside the authorized goal/handoff boundary;
- next action;
- stop rule that would force escalation.

Examples of autonomous decisions:

- choose the next pre-registered variant or signal family when the previous one
  failed honestly;
- skip or mark a bounded variant negative after debug evidence, then continue to
  the next planned variant;
- write a new preregistration artifact inside the same product/research goal,
  as long as final-OOT/test-set selection is not used;
- close a sub-route as negative and move to the next authorized sub-route;
- select cheaper CPU/file validation over GPU work when the handoff permits the
  validation.

Codex must stop with `DECISION NEEDED` only for hard decisions:

- changing the durable end goal or relaxing success criteria;
- using final OOT/test-set results to choose features, thresholds, models,
  strategy families, or claims;
- new online/paid data pulls, secrets, destructive cleanup, git operations, or
  resource escalation outside the handoff;
- no authorized next route remains after the documented autonomous decision
  budget is exhausted;
- repeated failure of the same class where continuing would be blind spin rather
  than a new bounded attempt.

CC's job at checkpoints is to audit these `AUTONOMOUS_DECISION` blocks. If they
are reasonable, do nothing. If they are not, CC writes a corrective
`CC_DECISION_*.md` or updated handoff and points Codex to it.

## Decision Continuation Protocol

When remote Codex stops at `DECISION NEEDED`, prefer continuing the same visible
session so the existing context stays alive. Do not inject a long multi-line
decision prompt directly into an already-running TUI. Multi-line `tmux
send-keys` can be split into partial UI commands and cause parser errors such as
`unknown command: do`.

Use this file-pointer pattern instead:

1. CC writes the decision as a structured file in the repo or run directory, for
   example `runs/<run>/CC_DECISION_<date>_<slug>.md`.
2. The decision file states the selected option, rationale, exact new/continued
   objective, files to read, success criteria, stop rules, and whether the
   current session should continue or a new session is required.
3. CC sends only one short line to the existing tmux TUI:

```text
CC/user decision recorded at runs/<run>/CC_DECISION_<date>_<slug>.md. Read it and continue this same session; keep updating runs/<run>/RUN_STATUS.md.
```

4. CC then polls `RUN_STATUS.md` and `tmux capture-pane` to confirm Codex read
   the decision and resumed work.

Start a new session only when the old session has exited, the TUI is visibly
polluted by a parser error, the decision changes the task so much that a clean
context is safer, or the old session ignores the one-line pointer. In that case,
the new `/goal` prompt must explicitly read the old `RUN_STATUS.md`, final
report, and the `CC_DECISION_*.md` file before acting.

## Monitoring And Correction

Default polling has two levels unless the user asks otherwise:

- Every 10 minutes: light decision check. Read `tmux ls`, recent
  `RUN_STATUS.md`, and recent `tmux capture-pane`; only decide whether there is
  `DECISION NEEDED`, a stalled/exited session, a boundary violation, or a user/CC
  decision requirement.
- Every 60 minutes: deep review. Read latest reports, broader `RUN_STATUS.md`,
  `git status -sb`, and enough pane history to judge convergence, direction
  quality, and whether CC should revise a handoff doc.

Polling should check:

- `ccusage` cost gate before starting new expensive CC-side actions.
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

If CC cannot reliably start an interactive remote session, it should stop after
audit/doc work and give the user an exact SSH command plus the exact `/goal`
prompt to paste. Manual launch is preferred over an invisible long-running
`exec` session when observability matters.

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

- Local-machine `ccusage` for CC/Claude Code crosses the user's configured
  24-hour threshold, currently USD 90. Do not apply this threshold as a remote
  Codex goal-session stop rule unless the user explicitly says so.
- Local, GitHub, and remote diverge in a way that cannot be safely fast-forwarded.
- A remote goal would require secrets to be printed, copied, committed, or pasted.
- The task would delete or move data, reports, runs, checkpoints, caches, archives,
  or credentials without explicit approval.
- The remote task exceeds the cost/scope/model limits written in the goal doc.
