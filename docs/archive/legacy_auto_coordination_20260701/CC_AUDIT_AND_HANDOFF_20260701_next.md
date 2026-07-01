# CC Audit & Active Codex Handoff (Next) — 2026-07-01

Author: CC (local Windows coordination/audit agent). Scope: ONE bounded, CPU-only
follow-up to the completed Track-C support-only closure + manuscript manifest.
`goal.md` remains the top steering authority. CC owns this doc; Codex executes it.
Supersedes the completed goal in `docs/CC_AUDIT_AND_HANDOFF_20260701.md`.

## Goal
Turn the existing CPU-only manuscript MANIFEST into a reviewer-ready DRAFT
PACKAGE, in place under
`reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/`,
by: (1) re-verifying that every manifest-referenced artifact exists and that
`manifest.json` is valid, and recording that verification as a checked artifact;
(2) expanding NARRATIVE_SKELETON.md into fuller draft prose sections; (3)
confirming the reproduction manifest is internally consistent and its ops scripts
exist; (4) producing a single reviewer-facing README/index that ties the
support-only closure to the pre-existing scaling-axis/failure-map package. NO new
GPU, NO training, NO checkpoint reads.

## Why now
The support-only branch is CLOSED (seed45 hard fail `support_pp_delta_below_0p04`;
predeclared no-hard-fail rule violated; not promotable). The remaining highest-
value, lowest-risk step is finishing the CPU manuscript package that already
exists as a manifest. Its inputs are verified complete (37 report artifacts, 10/10
figures hash-pass, 6 ops scripts present, reviewer checklist 6/6 pass), so this is
pure drafting + verification, not exploration. The multi-condition Track-C QUERY
route stays a separate, not-yet-launched hypothesis (needs its own split/gate/GPU
audit) and is out of scope here.

## Read first (exact server paths)
- goal.md
- docs/EXPERIMENT_INDEX.md, docs/PROJECT_REVIEW.md, docs/DECISIONS.md
- docs/CC_CODEX_COOPERATION_PROTOCOL.md (Strategic Escalation & Anti-Spin;
  Codex Goal-Doc Execution)
- reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/
  (REPORT_MANIFEST.md, REPRODUCTION_MANIFEST.md, NARRATIVE_SKELETON.md, manifest.json)
- reports/LATENTFM_SCALING_RESULT_SECTION_DRAFT_20260625.md
- reports/scaling_narrative_skeleton_20260625/result_sections.tsv
- reports/scaling_narrative_skeleton_20260625/reviewer_checklist.tsv
- reports/scaling_figure_readiness_20260625/figure_readiness.csv
- runs/latentfm_trackc_support_only_robustness_20260624/
  xverse_trackc_support_pairtype_both_train_multi_gene_resfilm_ep050_replay2_2k_seed45/RUN_STATUS.md

## Codex owns
- runs/<run>/RUN_STATUS.md for this task; all NEW/UPDATED files WITHIN
  reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/
  (expanded narrative, verification report, reviewer README/index). Codex may
  update manifest.json only to register newly added files in this same dir.

## CC owns
- goal.md, docs/EXPERIMENT_INDEX.md, docs/PROJECT_REVIEW.md, docs/DECISIONS.md,
  this handoff doc, and ALL git operations (add/commit/push/pull).

## Permissions
sandbox = workspace-write; CPU-only; model gpt-5.5; model_reasoning_effort = high.

## Forbidden
- No new GPU job; no training, inference, or checkpoint/weight reads.
- No reading canonical multi / held-out Track-C query data; no dataset edits.
- No destructive ops on datasets/checkpoints/runs/reports; do NOT delete or
  overwrite existing closure/negative-evidence files or the June 25/26 scaling
  artifacts (only ADD/EXPAND within the 20260701 manifest dir).
- No secret print/copy. No git commit/push/pull — CC manages git.
- No claim overreach: do NOT write "deployable monotonic scaling law",
  "checkpoint promotion", "Track-C query solved", "support-only promoted", or
  "chemical scaling success". Keep "leakage-safe scaling-axis audit / failure
  map"; default remains xverse_8k_anchor.

## Success criteria (measurable)
1. VERIFICATION artifact written
   (reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/VERIFICATION_REPORT.md
   + a machine-checkable list) showing: manifest.json re-validated (python -m
   json.tool); every manifest-referenced path re-checked exists=yes with a
   printed OK/MISS table; figure set re-confirmed 10/10 pass from
   figure_readiness.csv; missing count = 0.
2. NARRATIVE draft: NARRATIVE_SKELETON.md expanded (or a companion
   NARRATIVE_DRAFT.md added) with >= 6 drafted result sections in prose,
   one per row of scaling_narrative_skeleton_20260625/result_sections.tsv, each
   carrying its allowed-claim wording and explicitly avoiding its forbidden claim,
   plus the support-only closure section. Each section cites its primary artifact
   path(s).
3. REPRODUCTION consistency: confirm each ops script in the reproduction
   commands table exists; record any mismatch. State clearly the package is
   assembled-from-existing, not rerun.
4. REVIEWER INDEX: a single README/index (README_REVIEWER.md in the manifest
   dir) that, for a reviewer opening the folder cold, links closure evidence ->
   scaling package -> figures -> claims/limits, with the manuscript-safe claim and
   the forbidden-claim list up front.
5. RUN_STATUS.md records what was drafted/verified and the final output paths;
   final summary via codex --output-last-message.

## Stop rules (incl. anti-spin DECISION NEEDED)
- If re-verification finds ANY manifest-referenced artifact MISSING, a figure
  failing QC, or manifest.json invalid: STOP, do not paper over it. Append a
  `DECISION NEEDED` block to RUN_STATUS.md (what is missing, why it matters, 1-2
  options, the specific question for CC) and escalate.
- If a re-scored metric contradicts the closure (e.g. seed45 no longer reads as a
  hard fail): STOP and escalate with a DECISION NEEDED block; do NOT silently
  reopen or re-close the branch.
- ANTI-SPIN: if two substantive drafting/verification attempts do not measurably
  advance the success criteria, or the same failure class repeats twice, STOP and
  append a DECISION NEEDED block for CC. Do not keep burning cycles.

## Expected output paths
- runs/<run>/RUN_STATUS.md
- reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/VERIFICATION_REPORT.md
- reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/NARRATIVE_DRAFT.md
- reports/trackc_support_only_closed_scaling_manuscript_manifest_20260701/README_REVIEWER.md
- (optional) updated manifest.json registering the new files in this dir

## Progress reporting format
- Start with a brief plan in runs/<run>/RUN_STATUS.md.
- Append dated lines to RUN_STATUS.md as each success criterion is met (verify /
  narrate / reproduce-check / reviewer-index), each with the exact output path.
- End with a final summary via codex --output-last-message.
- Do NOT git commit/push/pull — CC manages git. Propose CC-owned doc updates
  (DECISIONS.md / EXPERIMENT_INDEX.md) as text in RUN_STATUS.md.
