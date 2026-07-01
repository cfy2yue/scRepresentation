# Decisions

Last slimmed: 2026-07-01.

The full pre-slim chronological decision log is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/DECISIONS.md
```

This file records only decisions that should guide new agents.

## 2026-07-01: RESULT — Scaling-Unit BLOCKED (needs per-arm geometry); Zebrafish Regularizer NEGATIVE; Manuscript VERIFIED

- SCALING-UNIT (run `scaling_unit_cpu_regression_20260701`): STOPPED at the hard
  prerequisite (anti-spin working). Existing artifacts collapse the best runs to one
  parent geometry, so the info/HVG-weighted-vs-cell-count regression cannot be fairly
  evaluated yet; no scaling-law claim allowed. Documented what's missing
  (`reports/scaling_unit_regression_20260701/missing_per_arm_artifacts.csv`,
  `per_arm_geometry_rows.csv`). NEXT (bounded CPU): materialize per-arm geometry
  (Vendi effective-count, effective rank, abundance/response-energy-weighted G_eff)
  from existing `*_pert_means.npz`, then rerun the regression vs cell count.
- ZEBRAFISH REGULARIZER (run `zscape_regularizer_mining_20260701_123950`): DECISIVE
  NEGATIVE — no candidate dynamic-response regularity generalized past the
  wrong-time/wrong-lineage/permutation nulls, so NO validated differentiable flow
  regularizer emerges (specs are diagnostic/rejected templates only). Confirmed a
  future expression-space prior must attach at `CoupledFM/model/train.py` `x1_hat/x_gt`
  (latent-only specs -> `v_pred` / `ode_integrate_diff`). Report:
  `reports/zscape_regularizer_mining_20260701_123950/LATENTFM_ZSCAPE_DYNAMIC_REGULARIZER_LAWS_20260701.md`.
  The negative PROTECTS the model from a false constraint; do NOT launch a regularizer
  from this coverage.
- MANUSCRIPT (run `cpu_manuscript_package_review_20260701_next`): CPU package verified
  reviewer-ready — manifest.json valid, 46/46 paths OK, 10/10 figures pass, 14/14 repro
  scripts, NARRATIVE_DRAFT + README_REVIEWER + VERIFICATION_REPORT added.

Both insight goals returned honest, bounded outcomes; default model stays
xverse_8k_anchor. Scaling next step is per-arm geometry materialization (CPU);
zebrafish regularizer is not supported on current coverage.

## 2026-07-01: REPRIORITIZE — Insight-Driven Scaling-Unit + Zebrafish Flow-Regularizer Over Flow-Matching Tuning

Decision: elevate two analysis-first, insight-driven scLatent threads above direct
flow-matching metric tuning (per user steering): (1) find the correct single-cell
SCALING UNIT — an information / effective-state axis (Vendi N_eff) and an
abundance/response-energy-weighted effective-gene-count G_eff, tested by CPU
regression over existing runs vs cell count; (2) mine the zebrafish wild-type
developmental reference atlas for GENERALIZABLE dynamic-response geometric laws (L2
developmental-tangent split, L1 state preservation) and spec each as a DIFFERENTIABLE
flow-matching regularizer. Both are CPU-only, decisive (win or publishable negative).

Reason: every raw-count scaling axis is closed (non-monotonic / no-harm-fail); the
HVG thesis is half-validated (concentration real; HVG-specific collapses to
abundance). Zebrafish is a rare GT for perturbation dynamics; UCE/species-latent is
closed, so mine geometry in expression + encoder-agnostic latent and constrain the
flow structurally. Architecture audit confirms the velocity field admits such a
regularizer cleanly, but an expression-space prior must attach in the raw-expr
trainer (no latent->gene decoder); it also found two metric defects (P4 eval MSE
random pairing; P1 aux 1-step vs eval 20-step) recorded for a later hygiene pass.

Consequence: historical route notes archived at
`docs/archive/legacy_auto_coordination_20260701/CC_AUDIT_AND_HANDOFF_20260701_scaling_unit.md` and
`docs/archive/legacy_auto_coordination_20260701/CC_AUDIT_AND_HANDOFF_20260701_zebrafish_regularizer.md`; architecture record
`docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md`. De-prioritized: flow-matching
endpoint tuning. Closed (do not reopen): UCE/species-latent; Track-C support-only GPU.

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

Consequence: historical goal/handoff note archived at
`docs/archive/legacy_auto_coordination_20260701/CC_AUDIT_AND_HANDOFF_20260701.md`.
Use current `local_goal.md`, `local_audit.md`, and `local_suggestion.md` for
new remote execution.

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
