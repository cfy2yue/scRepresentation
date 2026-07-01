# Prompt Library

These prompts are meant to be copied into other sessions or tools. They keep
project cleanup, audit, and handoff work consistent without forcing every agent
into the same exact execution path.

Available prompts:

- `SCLATENT_LOCAL_AUDIT_AND_GOAL_PROMPT.md`: ask a local model to understand
  scLatent, audit documents/results, and propose concrete next directions for
  the local audit packet.
- `CODEX_SCLATENT_COORDINATOR_PROMPT.md`: ask a Codex session to resume
  scLatent coordination after the user manually ends local audit.
- `CELLCLIP_DIRECTORY_ORGANIZATION_PROMPT.md`: ask the CellClip session to
  organize its own project and document data dependencies on scLatent.
- `STOCK_DIRECTORY_ORGANIZATION_PROMPT.md`: ask the stock session to organize
  its own project and remove redundant clutter safely.

Legacy local/remote auto-coordination prompts are archived under
`prompts/archive/legacy_auto_coordination_20260701/`. Do not use them for new
work. Current remote execution should be driven by repository-root
`local_goal.md`, `local_audit.md`, and `local_suggestion.md`. Remote execution
is active only when `local_goal.md` contains a filled `Exact Next Task`; use
`本地审计指令` and `本地审计结束` as the manual boundary triggers documented in
`docs/START_HERE.md`.
