# CC/Codex Cooperation Protocol

Updated: 2026-07-01

This file is the durable cooperation protocol for the scLatent repository. It
should be read by CC before local work and included in remote Codex handoff
prompts when server execution is requested.

## Ownership

- CC/Cursor owns local coordination: audits, goal refinement, docs, prompts,
  code review, small safe local patches, GitHub sync, and user-facing planning.
- Remote Codex owns server execution: implementation on the server, resource
  audits, experiments, long jobs, result integration, and progress reports from
  exact server paths.
- Do not edit the same code file from CC and remote Codex at the same time.
  Record ownership in `goal.md`, `docs/PROJECT_REVIEW.md`, or a dated handoff.

## Required Startup Check

Before non-trivial work:

```powershell
git -C E:\cc_workspace\scLatent fetch origin --prune
git -C E:\cc_workspace\scLatent status -sb
git -C E:\cc_workspace\scLatent rev-list --left-right --count HEAD...origin/main
ssh cyx-server-proxy-cfy "cd /data/cyx/1030/scLatent && git fetch origin --prune && git status -sb && git rev-list --left-right --count HEAD...origin/main"
```

If local, GitHub, and server differ, sync first or report the divergence. Do not
start broad edits from an old base.

## Local Scope

Allowed locally:

- read/review docs and source;
- audit stale paths, split boundaries, leakage risks, and handoff clarity;
- refine goals, prompts, and Markdown plans;
- prepare small doc/code patches when the user asks.

Not local by default:

- GPU jobs, model training, checkpoint evaluation, large data reads, or
  server-cache workflows;
- modifying `runs/`, `reports/`, `logs/`, checkpoints, pretrained weights,
  datasets, or credentials.

## Remote Codex Scope

When server work is needed, CC should hand remote Codex a concrete prompt with:

- objective and success criteria;
- files already inspected;
- files/tasks Codex owns;
- files/tasks CC owns;
- permissions and stop rules;
- expected output paths;
- whether to use `read-only`, `workspace-write`, or a stronger mode.

Use `gpt-5.5` for hard implementation/research planning and `gpt-5.4-mini` for
cheap status/doc checks. Current remote CLI order is `codex -a never exec ...`.

## Git Rule

Document locally, commit locally, push to GitHub only when requested or needed
for server sync. After push, update the server with:

```bash
cd /data/cyx/1030/scLatent
git pull --ff-only
```

## Strategic Escalation & Anti-Spin (added 2026-07-01)

Practice-learned division of judgment: Codex reliably handles code/training/
execution, but can lose strategic clarity or loop inside one direction. CC's
primary value is strategic audit and course-correction — not re-running compute.

- **Labor split (judgment):** Codex owns execution and local tactical choices. CC
  owns strategic direction, gate/stop-rule design, and course-correction. When a
  direction is ambiguous or not converging, treat it as a *strategy* bottleneck
  (CC's job), not a reason for more compute.
- **Anti-spin rule (Codex):** if two substantive attempts do not move measurably
  toward the goal's success criteria, or the same failure class repeats twice,
  **STOP**. Append a `DECISION NEEDED` block to `RUN_STATUS.md`: (1) what was
  tried, (2) what failed and why, (3) 1–2 concrete strategic options, (4) the
  specific question for CC. Do not keep burning compute on a stuck direction.
- **Escalation triggers (Codex → CC):** strategic ambiguity; repeated
  non-convergence; success criteria look unreachable within the cost/resource
  plan; scope creep; or a result that contradicts the goal's stated hypothesis.
- **CC cadence:** low-frequency (~1h) *strategic* check, not mechanical log
  polling. CC reads `RUN_STATUS.md` + codex last message, judges convergence and
  strategic soundness, and intervenes only to correct.
- **Corrective handoff (CC → Codex):** never silently mutate a running goal. CC
  writes a new dated subsection in the dated handoff doc (revised bounded goal,
  adjusted gate, pivot, or close) and re-hands it. Preserve negative evidence.

## Multi-Project Audit Workflow (added 2026-07-01)

When several projects need audit/optimization at the same time, CC should NOT
audit them serially in one shared context. Instead:

- Spawn ONE dedicated subagent session per project to run that project's audit /
  optimization independently (isolated context).
- The main CC session aggregates the subagent results, makes the cross-project
  priority calls, and performs the git commits/pushes and Codex handoffs.
- This avoids context confusion and cross-project bleed. Keep each subagent scoped
  to a single project; keep commits and handoffs centralized in the main session.

## Codex Goal-Doc Execution (preferred, added 2026-07-01)

Preferred handoff pattern: CC maintains the goal INSIDE the repo (`goal.md` and the
dated `docs/CC_AUDIT_AND_HANDOFF_<date>.md`), version-controlled. When launching
remote Codex, use goal mode and point Codex directly at that in-repo goal doc to
execute it - rather than maintaining a separate throwaway prompt. The launch prompt
should be a thin pointer, e.g.: "Read and execute the goal in
docs/CC_AUDIT_AND_HANDOFF_<date>.md; honor its ownership, stop rules, and success
criteria; start with a brief plan in runs/<run>/RUN_STATUS.md." This keeps a single
version-controlled source of truth. If goal-tracking is needed, enable it first
(`codex features enable goals`) and have Codex `/plan` to form a measurable goal
before executing.

## Default CC/Codex Role Model (added 2026-07-01)

- CC is the local coordinator and thinking layer: audit, critique, cheap local
  exploration, goal refinement, handoff docs, commits, pushes, and monitoring.
- For multi-project work, main CC should launch one independent CC subagent per
  project. Each subagent audits exactly one project and reports findings; it does
  not push, launch remote jobs, or edit outside its scope unless explicitly
  delegated.
- Main CC synthesizes subagent outputs, resolves contradictions, edits `goal.md`
  and dated handoff docs, then syncs through GitHub.
- Remote Codex is the remote execution layer: one approved remote task should get
  one dedicated goal session, usually one `tmux` session, with progress written
  to `runs/<run>/RUN_STATUS.md` or the task's equivalent status path.
- The same model should scale beyond the current three projects and beyond one
  server. Add new projects/servers to the registry first, then reuse the same
  subagent-audit and remote-goal-session workflow.

## Standard Task Processing Flow (default, solidified 2026-07-01)

Division of labor: **audit / direction / strategy judgment + planning + writing goal
docs = CC's job.** Codex only EXECUTES well-defined goals CC writes (implementation,
experiments, stats, training). Do NOT hand audit/strategy/direction judgment to Codex.

Every task follows this pipeline; multiple tasks run it IN PARALLEL — one subagent per
task (isolated to one repo), main CC coordinates + interfaces with the user; never
serialize independent tasks:
1. **Sync** — confirm the local git repo is current (fetch/status/rev-list; reconcile if
   diverged) before editing.
2. **CC audit + plan (local)** — audit strategy/defects/direction, decide the bounded
   next goal.
3. **(Optional) grounding** if the plan is uncertain — light checks/experiments via self
   SSH (read-only/small); HEAVY experiments via remote Codex, results fed back to CC.
4. **Finalize goal doc + audit doc → git commit + push** (secret scan + `diff --check`).
5. **Remote: git pull + launch Codex in goal-doc mode** via the 1030 wrapper
   `/data/cyx/1030/software/bin/codex` (preflight: `CODEX_HOME=/data/cyx/1030/software/codex_home`).
   Prefer a sidebar/resume-visible session IF it doesn't hurt efficiency/effect.
6. **Poll ~1h** — tmux/RUN_STATUS/last-message/convergence; anti-spin correct via a dated
   handoff.

**Goal mode + concrete DONE conditions (added 2026-07-01):** Prefer TRUE goal mode (the
`goals` feature is stable/enabled): interactive `codex` TUI in tmux (PTY) + `/goal`
pointing at the goal doc → codex keeps pushing until the goal is met, resumable via
`codex resume`, sidebar-visible. (`codex exec` is one-shot — completes bounded goals but
is not persistent/visible.) If using `exec`, the launch prompt MUST say "do not stop until
the concrete success criteria/thresholds are met; iterate within bounds — stop only for a
genuine evidence-based blocker/honest-negative, not after a shallow first pass." **Every
goal doc MUST state concrete numeric targets/thresholds + the exact DONE condition** so
codex can tell when it is finished; anti-spin/DECISION-NEEDED is for genuine blockers, not
early stopping.
