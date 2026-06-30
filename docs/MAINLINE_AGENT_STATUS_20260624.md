# Mainline Agent Status: 2026-06-24

## 2026-06-24 16:16 CST

## 2026-06-24 16:20 CST

Decision: both internal-passed scaling refill checkpoints failed frozen
canonical no-harm, and the OT/no-OT random rerun failed. The coordinator still
launched one bounded fourth-card exploratory smoke, `cap60_noot_3k_seed42`,
based on Raman's slate and the user's compute-first exploration preference.

Active jobs and resources:

* `lfm_xverse_scaling_cap60_noot_3k_seed42`: active on physical GPU2.
* Launch audit passed: RAM about `480 GiB` available, load1/core `0.142`,
  hard cap `max_user_gpus=4`, assigned GPU2.
* No other LatentFM tmux session was active at the one-time launch sanity check.

Macro progress:

* Current deployable/default remains `xverse_8k_anchor`.
* Scaling refill canonical no-harm failed for replay05 and seed42 6k.
* OT random rerun failed: `all_done_no_pass`, cross-bg pp delta `-0.016667`.
* Running smoke: cap60 x no-OT interaction on the leakage-safe cap60 primary19
  split, `OT_PAIR_MODE=random`, 3k steps, train-only internal gate.

Next legal path:

* Do not poll the no-OT interaction log before cadence unless exit evidence
  appears.
* If internal gate fails or is weak, close the no-OT interaction branch.
* If internal gate unexpectedly passes strongly, require a fresh explicit
  decision before any frozen canonical no-harm veto. Canonical multi and Track C
  query remain forbidden.

Blocker: no other fourth-card candidate is currently authorized. Seed44 is
blocked by seed43 failure; matched dataset-count and bootstrap target-noise are
closed by prior gates.

## 2026-06-24 16:16 CST

Decision: scaling refill produced a real train-only internal partial pass, so
the coordinator launched the predeclared frozen canonical no-harm veto jobs for
the two internal-passed checkpoints. This is not a promotion claim.

Active jobs and resources:

* `lfm_xverse_otpair_random_2k_seed42`: OT/no-OT random rerun posthoc pending.
* `lfm_scaling_ht_canon_xverse_scaling_cap60_replay05_4k_seed42`: frozen
  canonical no-harm veto, physical GPU2.
* `lfm_scaling_ht_canon_xverse_scaling_cap60_6k_seed42`: frozen canonical
  no-harm veto, physical GPU4.
* Last launch audits passed with RAM above `470 GiB` available and load/core
  well below the project CPU safety threshold. LatentFM remains within the
  4-physical-GPU cap and 48-core project budget.

Macro progress:

* Current deployable/default remains `xverse_8k_anchor`.
* Scaling refill internal decision:
  `/data/cyx/1030/reports/LATENTFM_SCALING_HIGH_THROUGHPUT_SMOKES_REFILL_DECISION_20260624.md`.
  Status `internal_partial_pass`.
* Internal passes: `xverse_scaling_cap60_6k_seed42`
  (`+0.012262/+0.016746/-0.000574`) and
  `xverse_scaling_cap60_replay05_4k_seed42`
  (`+0.010321/+0.012986/-0.000933`).
* Internal fail: `xverse_scaling_cap60_6k_seed43`
  (`-0.011650/-0.006940/+0.001662`), so seed44 confirmation is blocked.
* Canonical no-harm outputs are pending:
  `/data/cyx/1030/reports/LATENTFM_SCALING_HIGH_THROUGHPUT_CANONICAL_NOHARM_REFILL_REPLAY05_DECISION_20260624.md`
  and
  `/data/cyx/1030/reports/LATENTFM_SCALING_HIGH_THROUGHPUT_CANONICAL_NOHARM_REFILL_SEED42_DECISION_20260624.md`.
* Raman subagent (`019ef8b2-e6dd-7920-9ee0-e320a09bebde`) is auditing a
  possible fourth-card refill slate. It is read-only and must not launch jobs.

Next legal path:

* Wait on canonical no-harm exit markers by cadence, not log polling.
* If OT/no-OT random decision passes/near-passes, launch the guarded cap60
  no-OT interaction smoke.
* If Raman proposes a genuinely new, leakage-safe bounded candidate not covered
  by closed branches, launch it after fresh repeated resource audit.
* Do not relaunch matched dataset-count scaling or bootstrap target-noise;
  both already have fail/no-gpu or canonical-negative evidence.

Blocker: a fourth physical GPU is likely available, but no non-duplicate GPU
candidate is currently authorized while OT is pending and Raman is auditing the
next slate.

## 2026-06-24 15:02 CST

Decision: completed the next legal continuation as a CPU-only risk-stratified
gate, using corrected adjudication as current state. The gate status is
`risk_stratified_gate_fail_no_gpu`, so no GPU launch or frozen canonical
no-harm is authorized.

Active jobs and resources:

* `tmux ls`: no tmux server running.
* Active `cyx` LatentFM GPU use: none.
* GPU snapshot: GPU0 `1186 MiB`, util `22%`, from unrelated non-LatentFM
  activity; GPUs 1-7 about `27-28 MiB`, util `0%`. This is a one-shot snapshot,
  not a launch audit.
* RAM: about `480 GiB` available.
* CPU: non-LatentFM `cyx` MATLAB process PID `791779` is CPU-heavy at roughly
  `2337%`; LatentFM is not currently consuming CPU. Any later LatentFM launch
  must stay within the 48-core project cap and avoid total-host overload.

Macro progress:

* Current best deployable Track A setting remains `xverse_8k_anchor`.
* Risk-conditioned four-arm portfolio is closed as `mutate_not_promote`;
  canonical allowed `False`.
* Tian-Norman remains positive mechanism evidence only. Corrected adjudication
  removed the target-harm-row failure; remaining blocker is broad non-target
  risk-dataset harm.
* CPU-only risk-stratified gate report:
  `/data/cyx/1030/reports/LATENTFM_RISK_STRATIFIED_GATE_20260624.md`.
  Target strata pass (`TianActivation`, `NormanWeissman2019_filtered`), but
  non-target strata fail: Nadig hepg2 and Replogle RPE1essential fail
  mean-MMD/severe-row/top20-CVaR criteria; Nadig jurket and
  ReplogleWeissman2022_K562_gwps fail severe-row criteria.

Next legal path:

* No scalar gamma/replay continuation and no canonical rescue.
* A future launch would require either a distinct documented hypothesis or a
  default-off risk-row CVaR/top-k MMD code/unit gate, then fresh repeated GPU
  audit, RUN_STATUS, promotion gate, and stop rule.

Blocker: GPUs are idle for LatentFM, but there is no currently authorized
bounded GPU hypothesis under the corrected branch state.

## 2026-06-24 14:58 CST Coordinator Override

Decision: Peirce audit completed and the coordinator fixed the real
zero-as-missing summarizer gate bug. The corrected risk-conditioned decision is
`mutate_not_promote`; canonical allowed is `False`. Do not continue as if the
audit were pending, and do not launch scalar gamma/replay extensions of this
closed branch.

Active jobs:

* `tmux ls`: no tmux server running.
* Active `cyx` LatentFM GPU use: none.
* One-shot GPU snapshot: GPU1-7 about `27-28 MiB` and `0%` util; GPU0 about
  `1186 MiB` with low util from unrelated activity.

Resource/progress monitor note:

* Coordinator now explicitly owns resource-utilization monitoring and macro
  project-progress monitoring in `AGENTS.md`.
* A CPU-heavy non-LatentFM `cyx` MATLAB job is present; new LatentFM launches
  should still keep this project within the 48-core cap and avoid total-host
  overload.
* The next legal continuation is a CPU-only risk-stratified gate or a distinct
  documented hypothesis with fresh resource audit, RUN_STATUS, promotion gate,
  and stop rule.

## 2026-06-24 14:53 CST

Decision: accepted coordinator update that the risk-conditioned four-arm
portfolio completed and the batch summarizer has already run. I did not rerun
the summarizer and did not launch scalar gamma/replay sweeps. Status is
`risk_conditioned_internal_fail`, with the key conflict that the tian-norman
arm is aggregate-positive but failed row-tail criteria.

Active jobs:

* `tmux ls`: no tmux server running.
* Active `cyx` LatentFM GPU use: none.
* Other-user compute: physical GPU0 still has two `wly` Python compute
  processes; GPUs 1-7 appear empty by a one-sample resource snapshot. This is
  not a launch audit.

Resource snapshot:

* GPU snapshot: GPU0 `1186 MiB`, util `1%`; GPUs 1-7 about `27-28 MiB`, util
  `0%`.
* CPU/RAM: RAM available about `481 GiB`; top `cyx` CPU processes are Codex /
  VS Code infrastructure, no active LatentFM training/posthoc.
* Intended portfolio fit: `cyx` LatentFM GPU use is below the active-exploration
  target, but Peirce audit is pending at a decision conflict and the user
  explicitly forbade same-summarizer reruns and scalar gamma/replay sweeps.

Macro progress:

* Current best deployable Track A setting remains `xverse_8k_anchor`.
* Newly closed/blocked branch: risk-conditioned general-exposure mean-MMD/
  replay portfolio is internal-fail under current strict row-tail gate.
* Conflict detail: `tian-norman` has cross pp `+0.013525`, family pp
  `+0.018886`, family MMD `-0.001261`, Tian MMD `-0.010530`, Tian pp
  `+0.048974`, Norman MMD `-0.013571`, Norman pp `+0.151891`, but fails
  `target_harm_rows_too_many` / `risk_dataset_harm_count=5`.
* Non-overlapping preparation completed:
  `/data/cyx/1030/ops/synthesize_latentfm_tian_norman_harm_forensics_20260624.py`.
  Report:
  `/data/cyx/1030/reports/LATENTFM_TIAN_NORMAN_HARM_FORENSICS_20260624.md`.
  It decomposes the tian-norman row-harm failure using only completed
  train-only internal posthoc artifacts.
* Proposed next bounded hypothesis for Peirce/coordinator review only:
  default-off risk-row CVaR/top-k MMD over a predeclared risk dataset set, with
  CPU/unit validation first and only a capped 2k train-only smoke after external
  approval. This is a loss-design branch aimed at row tails, not a scalar
  gamma/replay sweep.

Next gate: wait for Peirce audit on whether the tian-norman conflict is a
gate-definition issue, true tail-risk blocker, or sufficient motivation for
the risk-row CVaR/top-k MMD code gate. No new GPU launch is authorized from
this status alone.

Blockers: external audit pending; no current non-duplicative GPU branch is
approved.

## 2026-06-24 14:50 CST

Decision: risk-conditioned general-exposure portfolio is not yet ready for the
batch summarizer. Marker-only check found `replayall`, `replaytian`, and
`noreplay` train/posthoc exit `0`; `tian-norman` train exit `0` with final
posthoc still running. I did not tail active logs.

Active jobs:

* `lfm_xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42` in tmux.
* Active `cyx` LatentFM GPU use: one posthoc eval on physical GPU1.
* Other-user compute: physical GPU0 has two `wly` Python compute processes and
  is not available for `cyx` planning.

Resource snapshot:

* `tmux ls`: one LatentFM tmux session, `lfm_xverse_general_exposure_tian_norman_mmd20_replayall_3k_seed42`.
* GPUs: GPU1 has the active `cyx` LatentFM posthoc; GPUs 2-7 appear empty by a
  one-sample resource snapshot, while GPU0 is occupied by another user. This is
  not a launch audit because no new GPU work was launched.
* CPU/RAM: RAM available about `479 GiB`; LatentFM posthoc CPU use about `152%`.
* Intended portfolio fit: current `cyx` GPU occupancy is below the active
  exploration target, but the active branch is at its terminal summary gate.
  Starting a fifth scalar/replay variant before this decision would duplicate
  the same branch without a new gate.

Macro progress:

* Current best deployable Track A setting remains `xverse_8k_anchor`.
* Closed promotion branches include high-throughput scaling, response repair,
  static latest rescue, soft exposure, OT/pairing, adaptive count, and scalar
  general-exposure MMD guard.
* Active branch: risk-conditioned general-exposure MMD/replay with
  dataset-filtered MMD hooks.
* Next gate: when `tian-norman` `POSTHOC_EXIT_CODE` is `0`, run
  `/data/cyx/1030/ops/summarize_latentfm_risk_conditioned_general_exposure_smoke_20260624.py`
  and inspect
  `/data/cyx/1030/reports/LATENTFM_RISK_CONDITIONED_GENERAL_EXPOSURE_SMOKE_DECISION_20260624.md`.
* Moving toward final best model: only conditionally. A risk-conditioned arm
  must pass the internal tail-risk gate, then receive external audit before any
  frozen canonical no-harm check. If it fails, this branch becomes negative
  mechanism evidence rather than a final-candidate path.

Next check gate: do not re-check the same long posthoc before about
2026-06-24 15:19 CST unless marker/tmux evidence shows natural completion.

Blockers: final `tian-norman` posthoc marker is pending; no new bounded,
non-duplicative GPU hypothesis is authorized before this branch decision.
