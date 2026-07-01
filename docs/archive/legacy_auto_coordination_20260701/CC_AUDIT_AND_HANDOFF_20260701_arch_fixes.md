# CC Audit & Active Codex Handoff — 2026-07-01 (LatentFM architecture optimization + validation)

Author: CC (from `docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md`). Engineering track:
implement + experimentally VALIDATE the top architecture fixes. CC owns direction/git;
Codex executes. Default model stays xverse_8k_anchor unless a fix clears a no-harm gate.

## GOAL (two bounded, validated fixes; do NOT stop until both validated OR documented)
**R1 — Fix eval velocity-MSE OT pairing (P4; eval-only, near-zero risk).** In eval Phase-1,
replace the two independent permutations of src/gt (`latent/train.py:3500-3501`) with OT
pairing (reuse `OTPrefetchIter`/`sinkhorn_pair`), so `test_mse` is comparable to `train_mse`.
- DONE(R1): on a FIXED existing checkpoint (no training), recomputed `test_mse` drops and
  tracks `train_mse` within a stable ratio across seeds; ODE-MMD/Pearson (pairing-free)
  UNCHANGED. CPU/eval-only.

**R2 — Align the auxiliary endpoint estimator with the eval integrator (P1/O2).** Behind a
config flag, replace the single-step `x1_hat = x_t + v·(1-t)` used by endpoint/direction/
composition losses with a short (4-step) `ode_integrate_diff` (already implemented).
- DONE(R2): on the xverse_8k_anchor recipe (endpoint5+comp006), across >=3 seeds INCLUDING
  the seed that hard-failed Track-C, held-out ODE-MMD/Pearson HOLD-OR-IMPROVE AND the
  endpoint/direction aux losses become monotone in the reported metric. Needs training.

## Why now
The architecture audit confirmed P4 (metric bug) and P1 (train/eval estimator mismatch,
likely feeding the seed-instability that closed Track-C, EXPERIMENT_INDEX:22). Fixing them
cleans model selection and likely reduces seed fragility — engineering foundation for the
scaling/zebrafish tracks. R3 (CFG condition dropout) is a stretch — only if R1+R2 land.

## Read first (server paths)
- docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md (P1-P10, attach points, top-3)
- CoupledFM/model/latent/train.py (eval Phase-1 ~3500-3501; aux losses ~2901-3195; ode_integrate_diff ~3225-3302; eval integrate ~3309-3348)
- CoupledFM/model/latent/fm_ot.py, CoupledFM/model/utils/data/ot_pairer.py (OT pairing to reuse)
- goal.md, docs/EXPERIMENT_INDEX.md, docs/DECISIONS.md, docs/CC_CODEX_COOPERATION_PROTOCOL.md (Standard Flow, GPU rules)

## Codex owns
CoupledFM/model/latent/ (the two fixes behind flags), runs/, RUN_STATUS.
## CC owns
goal.md, docs/EXPERIMENT_INDEX.md, docs/DECISIONS.md, docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md, ALL git.

## Permissions
sandbox = workspace-write; R1 = CPU/eval-only. R2 = GPU allowed but ONLY after a fresh
resource audit + a written RUN_STATUS (per scLatent GPU rule), <= the 4-GPU scLatent cap;
model gpt-5.5; effort high.

## Forbidden
- Launching GPU training without a resource audit + RUN_STATUS.
- Touching datasets/checkpoints destructively; deleting negative evidence; secret print/commit;
  git commit/push (CC finalizes). Changes must be behind flags (no silent default change).

## Do NOT stop until (goal-mode intent)
Keep working until BOTH R1 and R2 reach their DONE conditions with the multi-seed evidence,
OR you hit a genuine blocker (checkpoint/data missing, or R2 fails no-harm across seeds) →
then a DECISION NEEDED block with the per-seed table. Do not stop after a shallow pass;
do not flip the default without a no-harm gate.

## Success criteria (DONE)
R1 DONE (test_mse table vs train_mse across seeds; MMD/Pearson unchanged) AND R2 DONE
(>=3-seed held-out MMD/Pearson hold-or-improve + aux-loss monotonicity), each recorded in
RUN_STATUS.md with a proposed DECISIONS entry (keep-behind-flag vs promote-with-no-harm-gate).

## Expected outputs
runs/<run>/RUN_STATUS.md (R1 eval table, R2 per-seed table, resource audit) + code behind flags.

## Progress format
One line per step `[R1/R2 step] status | key metric`; final summary via --output-last-message:
R1 done? R2 done? per-seed numbers + keep/promote recommendation.
