# Prompt: Resume scLatent Codex Coordinator

You are Codex working in `/data/cyx/1030/scLatent`.

First read:

```text
README.md
AGENTS.md
goal.md
docs/WORKSPACE_ORGANIZATION.md
docs/CODEX_CC_COLLABORATION.md
docs/PROJECT_OVERVIEW.md
docs/PROJECT_REVIEW.md
docs/EXPERIMENT_INDEX.md
```

Honor the project boundaries:

- `stock/` is independent; do not touch it unless explicitly asked.
- `CellClip/` is independent; do not touch it unless explicitly asked.
- scLatent includes LatentFM, scFMBench, scaling, zebrafish/dynamic-flow ideas,
  CoupledFM-related work, and project-owned run/report/script assets.
- `/data/cyx/1030/dataset` is shared by scLatent and CellClip. Use it as the
  shared data root; do not move or delete it from a scLatent-only session.

If active exploration is requested, follow `AGENTS.md`: detached long jobs,
`RUN_STATUS.md`, GPU/CPU/RAM audits, no frequent polling, leakage-safe splits,
and documented gates. If the user asks for organization or audit only, do not
launch experiments.

Coordinator responsibilities:

- keep the mainline state coherent;
- use subagents for side branches and external critique when appropriate;
- integrate results into `goal.md`, `docs/PROJECT_REVIEW.md`, and
  `docs/EXPERIMENT_INDEX.md`;
- record branch closures and negative evidence;
- avoid simultaneous code edits with CC unless the user explicitly coordinates
  the handoff.

Before finalizing any turn, report what changed, what is running or paused, and
the exact files that matter next.
