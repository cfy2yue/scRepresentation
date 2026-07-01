# scLatent / scRepresentation Agent Rules

Use Chinese for user-facing summaries unless the user asks otherwise.

## Authority

Read these first:

- `goal.md`
- `local_goal.md`
- `local_audit.md`
- `local_suggestion.md`
- `remote_decision.md`
- `docs/START_HERE.md`

`goal.md` and the durable final goal in `local_goal.md` define the long-horizon
target. `Exact Next Task` is the current route hypothesis and priority start,
not the whole goal. If a route fails, record evidence and optimize the route;
do not mark the goal complete unless the final acceptance target is achieved.

## Remote Execution

- Long-run toward the final target until `ACHIEVED`, hard `BLOCKED`, or user
  interruption. `LOCAL_AUDIT_REQUEST` is a soft audit marker, not a stop reason.
- Remote Codex may make bounded `AUTONOMOUS_DECISION` choices inside the
  resource, split, data, and safety limits.
- If a route fails, metrics miss the target, or a local suggestion does not
  work, remote Codex should design a new safe route, optionally use subagents,
  record the decision in `remote_decision.md`, and continue.
- Remote Codex may launch subagents for independent audit, code review,
  provenance checks, metric review, or route pre-exploration. Subagents must
  read the same authority files, stay within project limits, and report
  evidence into RUN_STATUS/reports.
- Do not edit `local_goal.md`, `local_audit.md`, or `local_suggestion.md`
  during remote execution. Suggest next local updates in RUN_STATUS/reports.

## Project Boundaries

- Default deployable LatentFM state remains bounded by current docs until a
  strict no-harm/promotion gate supersedes it.
- Do not reopen Track-C support-only GPU work, held-out query/canonical-multi
  use, UCE/species-latent zebrafish, or GPU scaling replays unless
  `local_goal.md` explicitly authorizes the exact protocol.
- Do not claim a scaling law, validated regularizer, promoted checkpoint, or
  solved benchmark from collapsed, underpowered, or leakage-risk evidence.
- Generated runs/reports/checkpoints/caches stay server-local and out of Git
  unless explicitly curated as small documentation.

## Stop And Report

Output `LOCAL_AUDIT_REQUEST` when the route needs local strategy optimization,
required artifacts are missing, split/provenance controls are unclear, resource
boundaries would need to change, or results cannot be interpreted without
changing the final target.
Do not mark the long goal blocked for soft strategy issues, weak metrics, or a
failed suggested route. First record the evidence and route pivot in
`remote_decision.md`, then continue inside the approved resource/safety limits.
