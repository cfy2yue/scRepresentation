# Codex / CC Collaboration Protocol

Updated: 2026-06-30

## Roles

Codex main/coordinator:

- owns scLatent mainline execution and resource monitoring;
- launches experiments when active exploration resumes;
- integrates results into `goal.md`, `docs/PROJECT_REVIEW.md`,
  `docs/EXPERIMENT_INDEX.md`, and decision reports;
- handles exceptions, failed jobs, and branch closure decisions;
- keeps the project state coherent for future sessions.

Cursor/CC:

- is useful for read-only audit, literature/context synthesis, goal refinement,
  documentation cleanup, and independent critique;
- can propose directions, benchmark fixes, mechanism hypotheses, and prompt
  improvements;
- can edit docs when the user asks it to organize or refine project writing;
- should edit code only when the user explicitly says Codex is paused for that
  file/branch or asks CC to take over a scoped implementation.

## Avoiding Conflicts

Before CC edits code or shared project files, it should check:

```bash
pwd
ls -la
tmux ls
find runs -maxdepth 2 -name RUN_STATUS.md | tail
```

If running from the workspace root, CC should enter `/data/cyx/1030/scLatent`
before inspecting scLatent project files. `dataset/` at the workspace root is
shared between scLatent and CellClip; editing or deleting shared data requires
explicit user direction.

If Codex is actively running or editing the same branch, CC should stay
read-only and write an audit/report instead. If the user says Codex is paused,
CC may edit the requested files and must document exactly what changed.

For docs, concurrent work is lower risk but still needs care:

- prefer appending dated sections over rewriting long history files;
- avoid deleting old negative evidence;
- do not summarize away exact paths to reports, logs, and run statuses;
- make new recommendations actionable, with gates and stop rules.

## Writing Expectations

Both Codex and CC should write at the documentation level when asked to
coordinate the project:

- record key decisions;
- record important results and negative evidence;
- record implementation status and exact paths;
- explain why a branch is kept, promoted, demoted, or closed;
- keep next actions concrete.

Do not overfit the docs into a rigid script. The goal is to let the next agent
understand the state quickly and still use judgment.

## scLatent Canonical Docs

Use these as the main written state:

```text
goal.md
docs/PROJECT_REVIEW.md
docs/EXPERIMENT_INDEX.md
docs/DECISIONS.md
docs/RESULTS_SUMMARY.md
docs/BUGS_AND_FIXES.md
docs/WORKSPACE_ORGANIZATION.md
docs/CODEX_CC_COLLABORATION.md
```

## Recommended CC Output Shape

When CC is asked to audit or optimize the project, request:

```text
1. What files were read.
2. Current best understanding of the project state.
3. Suspected stale/contradictory documentation.
4. Strongest current evidence and weakest assumptions.
5. Concrete next directions, each with hypothesis, required files/scripts,
   resource needs, gate, and fail-close rule.
6. Exact doc edits proposed or made.
7. Any code files that should not be edited until Codex pauses.
```
