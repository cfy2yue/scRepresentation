# Prompt: CC Audit And Goal Optimization For scLatent

You are entering `/data/cyx/1030/scLatent`, the clean entrypoint for the
scLatent project. Most project assets now live physically under this folder.
The shared data root is `/data/cyx/1030/dataset`, and old reports may still
mention historical root paths.

Task: perform a read-first project audit and help optimize the goal/documentation
for the next Codex execution cycle. Do not launch training or GPU jobs. Do not
edit code unless I explicitly say Codex is paused for that file/branch.

Read first:

```text
README.md
AGENTS.md
goal.md
local_goal.md
local_audit.md
local_suggestion.md
docs/START_HERE.md
docs/WORKSPACE_ORGANIZATION.md
docs/PROJECT_OVERVIEW.md
docs/PROJECT_REVIEW.md
docs/EXPERIMENT_INDEX.md
docs/RESULTS_SUMMARY.md
docs/DECISIONS.md
docs/BUGS_AND_FIXES.md
```

Then inspect the most relevant recent reports in `reports/` and run statuses in
`runs/*/RUN_STATUS.md` that relate to:

- current best LatentFM model/settings;
- Track A benchmark/control-baseline status;
- Track C support/query split status;
- scaling-law and information-scaling experiments;
- zebrafish/dynamic perturbation/flow-matching inspiration;
- SciPlex pharmacogenomic source/source-control attempts;
- any branch recently closed by gates.

Output requirements:

1. List the exact files you read.
2. Summarize current project state in 10-20 bullets.
3. Identify stale, contradictory, or overgrown documentation.
4. State the strongest results and the most important negative evidence.
5. Propose 3-6 next directions. Each direction must include:
   - hypothesis;
   - why it is not a duplicate of a closed branch;
   - exact data/split/evaluation boundary;
   - required scripts or missing implementation;
   - resource plan;
   - promotion gate;
   - fail-close rule.
6. Propose concrete edits to `local_goal.md`, `local_audit.md`, and
   `local_suggestion.md`. Do not edit `goal.md` unless the user explicitly
   changes the durable final target.
7. Flag any code edits that should wait until Codex pauses.

Keep the audit sharp and actionable. The goal is to improve the next execution
cycle, not to write a broad essay.
