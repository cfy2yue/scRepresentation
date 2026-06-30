# Decision: Pause Pairwise Endpoint GPU Scale-Up

## Date

2026-06-22 19:23 CST

## Context

Track C endpoint A/B blocks closed 8/8 variants. C/D no-harm/head blocks then
closed 16/16 variants. C-block pairwise-condition endpoint runs showed the
strongest partial signal, with support pp deltas about `+0.0090` to `+0.0106`
and clean canonical family no-harm, but still failed the formal `+0.02` support
gate.

## External Review

A read-only subagent reviewed AGENTS, PROJECT_REVIEW, latest goal entries,
A/B and C/D decision summaries, and C/D manifests. It recommended not
continuing GPU scale-up yet because the best pairwise-condition support signal
is only about half the gate and C/D closed 16/16.

## CPU Diagnostic

`/data/cyx/1030/reports/LATENTFM_TRACKC_SUPPORT_SIGNAL_CEILING_DIAGNOSTIC_20260622.md`
showed:

* overall condition-level mean pairwise endpoint delta `+0.010519`;
* median delta `+0.001376`;
* Norman mean delta around `+0.018` to `+0.0218`;
* Wessels near zero or slightly negative;
* mean candidate-to-support-route pp gap `+0.565753`.

## Decision

Do not launch the prepared E-block GPU scale-up yet:

`/data/cyx/1030/ops/launch_latentfm_trackc_pairwise_endpoint_scale_e_20260622.sh`

## Reason

The evidence points to a route-transfer/conditioning bottleneck, especially on
Wessels, rather than a simple endpoint-weight or training-length shortage.

## Consequence

Next work should be CPU-only route-transfer diagnostics or a new mechanism gate
that explains how to transfer the fixed support route/memory readout signal
into the model before more Track C GPU expansion.

## Follow-Up Diagnostic

The route-transfer bottleneck diagnostic
`/data/cyx/1030/reports/LATENTFM_TRACKC_ROUTE_TRANSFER_BOTTLENECK_DIAGNOSTIC_20260622.md`
confirmed the pause decision:

* Norman average mean delta by run: `+0.019636`;
* Wessels average mean delta by run: `-0.000255`;
* weighted route-gap closure: `+0.037230` for Norman and `-0.000186` for
  Wessels.

The prepared E launcher remains disabled unless a new CPU gate shows a
mechanism that materially improves route-gap closure, especially on Wessels.
