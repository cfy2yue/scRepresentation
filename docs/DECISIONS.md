# Decisions

Last slimmed: 2026-07-01.

The full pre-slim chronological decision log is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/DECISIONS.md
```

This file records only decisions that should guide new agents.

## 2026-07-01: RESULT — Track-C Support-Only CLOSED; CPU-Only Manuscript Manifest Built

Decision: CLOSE the Track-C pair-type support-only branch. Gate evaluation (Codex):
seed43 pass (pp/MMD `+0.101347/-0.008834`), seed44 pass (`+0.073213/-0.004449`),
seed45 HARD FAIL (`+0.032864/-0.000943`, reason `support_pp_delta_below_0p04`).
2/3 seeds pass, but seed45 violates the predeclared no-hard-fail condition → not
promotable.

Reason: the predeclared gate requires 2/3 pass AND no hard fail; the hard fail
stands. Preserve the negative evidence and pivot to the CPU-only deliverable.

Consequence: manuscript-ready artifact assembled (CPU-only, no new GPU) at
`reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/`
(`REPORT_MANIFEST.md`, `REPRODUCTION_MANIFEST.md`, `NARRATIVE_SKELETON.md`,
`manifest.json`; JSON-validated, referenced paths exist). Next scLatent step is
manuscript polish, not further support-only GPU work. Multi-condition Track-C
query route remains a separate, not-yet-launched hypothesis.

## 2026-07-01: CC Audit — Close Track-C Support-Only If No-Hard-Fail Violated; Pivot To Manuscript

Decision: after a clean three-way sync (local = GitHub = server at `56a9bd2`),
hand remote Codex one bounded CPU-only goal — evaluate the predeclared Track-C
support-only 2/3-seed + no-hard-fail gate from the completed posthoc. If the
no-hard-fail condition is violated (seed45 hard-failed), close the support-only
branch with negative evidence preserved and assemble the existing CPU-only
scaling-axis/failure-map report into a manuscript-ready artifact. No new GPU.

Reason: a seed-level hard fail violates the predeclared no-harm gate, so the
branch is very likely not promotable. The highest-value deliverable now is the
CPU-only manuscript package, not further GPU exploration.

Consequence: see `docs/CC_AUDIT_AND_HANDOFF_20260701.md` for the full goal +
ownership and the anti-spin escalation rule (if seed44 posthoc passes cleanly and
seed45 re-scores non-hard-fail, escalate instead of closing). CC owns
goal/index/review/decision/handoff docs; Codex owns runs/reports/RUN_STATUS.

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
