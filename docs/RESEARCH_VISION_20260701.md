# scLatent Research Vision — 2026-07-01 (user-directed)

Author: CC, recording the user's stage goals verbatim-in-substance so they are not
forgotten. These are **scientific exploration** directions (insight-first); they may
or may not improve LatentFM's final prediction, but they aim to understand the true
biology/statistics and, where a real regularity is found, to CONSTRAIN the model.
Three tracks proceed in parallel: (A) Scaling, (B) Zebrafish discovery, (C) LatentFM
architecture optimization.

## A. Scaling — a quantitative single-cell scaling law + "information content"

Motivation: single-cell data is highly heterogeneous — **big scale can mean low
information / redundancy**. The external review
`external_review/scFM-dataselection` raises exactly this (data selection for scFMs).
Almost nobody has proposed a scaling LAW for single cell. Goal: define a **quantitative
scaling x** and a notion of **single-cell-set "information content"**, mainly on
**perturb-seq datasets**, to determine *what information is training-effective /
high-information-density* and *what training regime is reasonable* — then a quantitative
scaling law that can also guide LatentFM training.

Angles to develop (发散更多角度; imagination space is large):
- **Clustering view**: effective number of distinct states (not raw cells); Vendi /
  effective-rank / cluster-count at fixed inertia.
- **Pair view**: after clustering + OT, use PAIRS to measure information — e.g. **how
  many kinds of pair-modes/patterns** exist (pair-mode diversity as an information axis).
- **Statistical / information-theoretic**: entropy, MI, effective sample size (Kish),
  participation ratio; **abundance/response-energy-weighted effective gene count G_eff**
  (HVG concentration is real — top-2k ≈ 84% response energy — but the HVG-specific signal
  collapses to abundance, so weight by abundance/response-energy, not a bespoke HVG score).
- Note: current LatentFM already does scaling experiments (condition-count generalization,
  cell-background generalization, perturbation-condition generalization) but it is **not
  systematic and has produced no clear insight yet** — the goal is to systematize into a
  quantitative law.
- Immediate blocker (from the 2026-07-01 goal): existing runs collapse to one parent
  geometry; need PER-ARM geometry materialization before any regression is fair.

Deliverable target: a quantitative scaling-x definition + evidence it predicts model
performance better than cell count (held-out, LODO, abundance-confound-controlled), i.e.
"performance scales with effective information / distinct-state / pair-mode count, not
cell count" — and its implication for data selection + HVG budget + how much data helps.

## B. Zebrafish — scientific discovery of perturbation dynamics (from time-series GT)

This is a **scientific research/discovery** track. Understand the related work first:
- `ref/zebrafish_dataset.pdf` (no need to read fully) and Codex's synthesis
  `docs/literature/SCALING_ZSCAPE_SQUIDIFF_NOTES_20260701.md` — read+understand, then
  return to key spots of the PDF as needed.

Core question: from the precious **time-series** data, discover how cells respond to
perturbation — what is the dynamic process? Two lenses:

- **Distribution (macro) view**: across and between time points, look at distribution
  statistics and their regularities (e.g. **e-distance**, means, higher moments over
  time).
- **Individual view**: first use **OT to build single-cell-dimension time-series
  samples**. Getting single-cell info across snapshots is itself deep — it relies on
  pairing, but NOT just two time points: **multi-timepoint pairing**, sampling one cell
  per time stage to form a **pseudo single-cell tracking** trajectory. (Worth careful
  design.)
  - Individual analysis modalities:
    - **Expression space**: target-gene changes, marker-gene changes; introduce
      **CellOracle / GEARS**-style GRNs to test whether signal propagates from the target
      gene outward; pathway enrichment to see cross-pathway association and whether a
      single pathway shows an upstream→downstream cascade.
    - **Latent space**: geometry, direction (of the transition/flow).

Hypothesis: with so many angles there should be some regularities. They may not lift
LatentFM's final accuracy, but they are closer to the true state — and a real one could
be introduced as a **regularizer to make the trajectory more accurate**. (Prior
2026-07-01 mining found no regularity generalizing on the narrow coverage; this vision
broadens the search substantially — distribution view + pseudo-sc-tracking +
GRN/pathway-cascade + latent geometry.)

## C. LatentFM architecture optimization (engineering focus, parallel)

Optimize the architecture and **experimentally validate** each optimization — this is
the engineering priority, run in parallel with A/B. Grounded in the 2026-07-01 audit
(`docs/LATENTFM_ARCHITECTURE_AUDIT_20260701.md`): fix the eval-MSE OT-pairing defect
(P4) and the train/eval estimator mismatch (P1); address additive-only conditioning /
no-CFG (P2/P3) and the collinear-aux gradient conflict (P5, likely the seed-instability
root cause); the regularizer from track B attaches at the raw-expression trainer
(`CoupledFM/model/train.py` x1_hat/x_gt) since the latent trainer has no latent→gene
decoder. Each change = a bounded experiment with a success criterion.

## Sequencing
A, B, C proceed in parallel. A and B are analysis-first (CPU / ≤1 GPU); C is
experimental. Codex executes on the server (1030 account); CC audits/synthesizes and
owns goal/decision docs. Anti-spin + no-harm gating apply throughout.
