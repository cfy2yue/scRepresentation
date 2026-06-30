# Decisions

Last slimmed: 2026-07-01.

The full pre-slim chronological decision log is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/DECISIONS.md
```

This file records only decisions that should guide new agents.

## 2026-07-01: Use scRepresentation As The GitHub Repository

Decision: the active GitHub repository for `/data/cyx/1030/scLatent` is
`https://github.com/cfy2yue/scRepresentation`.

Reason: the server directory remains named `scLatent`, but the user clarified
that the publication/coordination repository should be `cfy2yue/scRepresentation`.

Consequence: `origin` should point to
`https://github.com/cfy2yue/scRepresentation.git`; CC should clone that repo
into a local folder named `scLatent`. Any earlier GitHub target using the old
server-directory name as the repository name is superseded for this workspace.

## 2026-06-30: Publish scLatent As A Monorepo

Decision: initialize `/data/cyx/1030/scLatent` as the project-level repository
for `https://github.com/cfy2yue/scRepresentation`.

Reason: scLatent is the top-level working project. `CoupledFM/` and
`scFMBench/` are currently treated as ordinary nested source directories, not
Git submodules, because the user said those two old repositories are
temporarily not needed and scLatent should be maintained as the umbrella repo.

Consequence: previous nested `.git` metadata was preserved under ignored
server-local backup, while source files remain available in the monorepo.

## 2026-06-30: Keep Shared Data At Workspace Root

Decision: keep shared data under `/data/cyx/1030/dataset`.

Reason: scLatent and CellClip both need this root. Moving it into either project
would break project boundaries and duplicate large assets.

Consequence: project docs and scripts should reference
`/data/cyx/1030/dataset/...` for shared data and
`/data/cyx/1030/scLatent/...` for scLatent-owned assets.

## 2026-06-30: Use Server-Local Runtime Symlink

Decision: expose the existing Conda install through
`/data/cyx/1030/software/miniconda3` as a symlink to
`/data/cyx/software/miniconda3`.

Reason: physically relocating a Conda install is risky; the symlink gives the
new workspace path without breaking the working environment.

Consequence: `init-scdfm.sh` uses the `/data/cyx/1030/software/miniconda3/...`
path, while the original install remains in place.

## 2026-06-30: Slim Git-Tracked History

Decision: replace very large chronological Markdown logs with short current
state/index documents and preserve full versions in ignored server-local
archive.

Reason: CC/Codex handoff needs a crisp entrypoint. Long intermediate logs are
valuable provenance but poor GitHub onboarding material.

Consequence: GitHub carries the current decision state; server archives preserve
full historical evidence.

## Standing Decision: No Heavy Outputs In Git

Do not track datasets, checkpoints, `runs/`, `reports/`, logs, local archives,
venvs, secrets, tokens, or large binary model/data artifacts. Track source code,
small configs, README/AGENTS, and high-signal docs.
