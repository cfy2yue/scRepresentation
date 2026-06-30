# Project Review

Last slimmed: 2026-07-01.

The full pre-slim review log is preserved on the server, outside Git, at:

```text
docs/local_archive/20260630_pre_slim/PROJECT_REVIEW.md
```

Use this file for the current decision state. Use the archive only when a
specific historical run, metric, or negative result needs provenance.

## Current Project Goal

scLatent is the server/GitHub monorepo for LatentFM-style latent perturbation
prediction, scFMBench evaluation infrastructure, CoupledFM-derived code, and
scaling-law/failure-map research. The near-term goal is to keep the project
reproducible, leakage-safe, and publication-ready while avoiding misleading
GitHub history dumps.

## Current Path

- Keep runnable code and high-signal docs in Git.
- Keep large generated evidence (`runs/`, `reports/`, checkpoints, logs,
  datasets, local archives) on the server and out of Git.
- Treat scaling results as evidence for a scaling-axis audit and failure map,
  not as a deployed model claim.
- Require a fresh resource audit, written hypothesis, leakage boundary, stop
  rule, and `runs/<run>/RUN_STATUS.md` before any future long/GPU run.

## Evidence Snapshot

- Existing historical reports support `xverse_8k_anchor` as the current default
  until a newer strict gate supersedes it.
- The 2026-06-25 scaling/NM artifacts produced provenance manifests, figure QA,
  claim/failure packages, and narrative skeletons via short CPU/report scripts.
- Those artifacts support manuscript organization and negative-evidence mapping,
  but do not authorize a new GPU branch without a fresh experiment plan.
- Root-level scLatent-owned assets have been consolidated under
  `/data/cyx/1030/scLatent`; shared data remains under `/data/cyx/1030/dataset`.

## Failure Modes And Risks

- Old docs can still contain root-level historical paths. Treat them as
  provenance unless they are current entry docs or runnable scripts.
- Large chronological logs in Git make CC/Codex handoff harder and increase the
  chance that a new agent follows obsolete instructions.
- Publishing datasets, reports, checkpoints, venvs, tokens, or logs would create
  size, privacy, and reproducibility problems.
- Claim language around scaling can overreach if it says "deployable scaling
  law" instead of "leakage-safe scaling-axis audit/failure map".

## Direction Decision

Continue with modification.

The codebase and evidence are worth preserving, but GitHub should carry a slim
current-state document set. Full historical logs remain server-local unless a
small, curated excerpt is needed for publication.

## Recommended Next Action

After this cleanup, push the slim initial GitHub state only if the user confirms
the remotes are correct. Do not force-push over non-empty remotes without an
explicit user decision.

For research continuation, create a new dated goal section with:

- hypothesis;
- split/evaluation boundary;
- resource plan;
- command;
- expected outputs;
- promotion gate;
- stop rule.

## Files To Inspect Next

```text
goal.md
docs/EXPERIMENT_INDEX.md
docs/DECISIONS.md
docs/BUGS_AND_FIXES.md
docs/WORKSPACE_ORGANIZATION.md
docs/GIT_AND_COLLABORATION.md
```
