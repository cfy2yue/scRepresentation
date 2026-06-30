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
