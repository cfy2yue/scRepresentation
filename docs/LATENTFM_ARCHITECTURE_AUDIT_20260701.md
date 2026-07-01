# LatentFM Architecture Audit — 2026-07-01

Author: CC (from a code-grounded architecture audit). Record of findings; not a
directive. The maintained model = `ControlMLPVelocityField`
(`CoupledFM/model/latent/models/mlp.py`), trained by
`CoupledFM/model/latent/train.py`. Raw-expression sibling
(`CoupledFM/model/train.py` + `model/models/velocity_field.py`) matters for the
expression-space regularizer.

## How LatentFM works (key file:line)
- Conditional flow-matching in a precomputed latent space (emb_dim 2058, xverse/stack;
  `latent/config.py:29,42`). Control->GT flow, linear CondOT interpolant
  `x_t=(1-t)x0+t x1`, target velocity `dx=x1-x0` (`latent/fm_ot.py:158-165`), loss
  `mse(v_pred, dx)` (`latent/train.py:2901`).
- Velocity net: shared encoder over `x_t,x_0`; conditioning `c = TimestepEmb(t) +
  ctrl_proj(h_0) + p_d`; N=8 adaLN-Zero residual blocks; zero-init output (identity
  flow at init) (`mlp.py:730-771,387-388`).
- Perturbation conditioning is ADDITIVE (`c += p_d`, `mlp.py:771`), no cross-attention;
  `pert_to_c` zero-init (`mlp.py:397-399`). Encoder pools gene-id embeddings
  (`condition_emb/genepert/perturbation_encoder.py`).
- Eval = explicit Euler ODE, 20 steps from src (`train.py:3309-3348,3665`). Optional
  differentiable Euler `ode_integrate_diff` (`train.py:3225-3302`).
- One condition per batch (bs 256), OT pairing in `OTPrefetchIter` before train_step
  (`train.py:2218-2358`, `utils/data/ot_pairer.py`). Default `xverse_8k_anchor` =
  FM+MMD+composition~0.06+endpoint~5.

## Potential problems (prioritized)
- **P1 (confirmed) train/eval estimator mismatch**: aux endpoint/direction/composition
  losses use a single Euler step `x1_hat=x_t+v(1-t)` (`train.py:2941,2958,2969,3003`)
  but eval integrates 20 steps — aux losses optimize a proxy.
- **P4 (confirmed) eval velocity-MSE random pairing**: eval permutes src/gt
  independently (`train.py:3500-3501`) instead of OT pairing -> `test_mse` not
  comparable to `train_mse`, biases model selection. (Headline ODE-MMD/Pearson are
  pairing-free, unaffected.)
- **P5 (partly probed) gradient conflict**: many collinear mean-delta aux objectives
  with static weights (`train.py:2901-3195`); plausibly the source of seed sensitivity
  that hard-failed Track-C seed45.
- **P2/P3 (confirmed) additive-only conditioning, no CFG**: field learns a
  near-condition-independent mean flow; support/Track-C adapters are compensations;
  a capacity bottleneck for unseen/combinatorial perturbations.
- **P6 (confirmed) batch-mean losses** collapse within-condition heterogeneity.
- **P7 (confirmed)** Euler-20 + linear-path, no curvature control; metrics depend on
  ode_steps.
- **P8-P10 (plausible)** OT cost in raw latent scale (anisotropy); chem/gene fusion
  hard-blend inconsistency; min-pool inf-fill under bf16.

## Regularizer attachment (for the zebrafish direction)
- Velocity field is EXPLICIT and accessible; `x_t,dx` available; `ode_integrate_diff`
  exposes a differentiable trajectory. EASY: penalize `v(x_t,t)`, one/multi-step
  endpoint, OT-coupling, straightness/curvature/speed — add as a fixed-weight aux term
  in `train_step` (mirror existing direction/endpoint pattern).
- HARD / key constraint: training is single-t per batch-mean (must add own multi-t
  eval for curvature); couplings are OT (a curvature prior must reconcile with OT);
  **the latent trainer has NO latent->expression decoder** (emb_dim 2058, latent-only)
  -> an EXPRESSION-space prior must attach in `CoupledFM/model/train.py` (gene-space
  `x1_hat`/`x_gt` at ~1643-1644) or requires adding a frozen decoder. This is the
  single biggest structural gap for the "regularize in expression AND latent space" plan.

## Optimization space / top-3 (recorded; NOT launched — user priority is insight-driven work)
- R1 (near-zero risk, metric-only): fix eval velocity-MSE to OT-pair src/gt.
- R2 (low risk): align aux endpoint estimator with the eval integrator (use a short
  `ode_integrate_diff`).
- R3 (low-moderate): CFG-style condition dropout (~10%) + inference guidance scale.
- Also: gradient-surgery/uncertainty weighting (promote the existing PCGrad no-harm
  gate); FiLM/cross-attention conditioning; reflow/straightness reg (ties to the
  zebrafish prior); OT cost robustification (zscore_l2/Hungarian).

These become a separate optional "architecture hygiene" goal if/when the user
greenlights it; they are documented here so the insight goals can reference the exact
attachment points and the P4/P1 metric caveats.
